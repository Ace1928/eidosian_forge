from .interpolatableHelpers import *
from fontTools.ttLib import TTFont
from fontTools.ttLib.ttGlyphSet import LerpGlyphSet
from fontTools.pens.recordingPen import (
from fontTools.pens.boundsPen import ControlBoundsPen
from fontTools.pens.cairoPen import CairoPen
from fontTools.pens.pointPen import (
from fontTools.varLib.interpolatableHelpers import (
from itertools import cycle
from functools import wraps
from io import BytesIO
import cairo
import math
import os
import logging
def draw_glyph(self, glyphset, glyphname, problems, which, *, x=0, y=0, scale=None):
    if type(problems) not in (list, tuple):
        problems = [problems]
    midway = any((problem['type'] == 'midway' for problem in problems))
    problem_type = problems[0]['type']
    problem_types = set((problem['type'] for problem in problems))
    if not all((pt == problem_type for pt in problem_types)):
        problem_type = 'mixed'
    glyph = glyphset[glyphname]
    recording = RecordingPen()
    glyph.draw(recording)
    decomposedRecording = DecomposingRecordingPen(glyphset)
    glyph.draw(decomposedRecording)
    boundsPen = ControlBoundsPen(glyphset)
    decomposedRecording.replay(boundsPen)
    bounds = boundsPen.bounds
    if bounds is None:
        bounds = (0, 0, 0, 0)
    glyph_width = bounds[2] - bounds[0]
    glyph_height = bounds[3] - bounds[1]
    if glyph_width:
        if scale is None:
            scale = self.panel_width / glyph_width
        else:
            scale = min(scale, self.panel_height / glyph_height)
    if glyph_height:
        if scale is None:
            scale = self.panel_height / glyph_height
        else:
            scale = min(scale, self.panel_height / glyph_height)
    if scale is None:
        scale = 1
    cr = cairo.Context(self.surface)
    cr.translate(x, y)
    cr.translate((self.panel_width - glyph_width * scale) / 2, (self.panel_height - glyph_height * scale) / 2)
    cr.scale(scale, -scale)
    cr.translate(-bounds[0], -bounds[3])
    if self.border_color:
        cr.set_source_rgb(*self.border_color)
        cr.rectangle(bounds[0], bounds[1], glyph_width, glyph_height)
        cr.set_line_width(self.border_width / scale)
        cr.stroke()
    if self.fill_color or self.stroke_color:
        pen = CairoPen(glyphset, cr)
        decomposedRecording.replay(pen)
        if self.fill_color and problem_type != InterpolatableProblem.OPEN_PATH:
            cr.set_source_rgb(*self.fill_color)
            cr.fill_preserve()
        if self.stroke_color:
            cr.set_source_rgb(*self.stroke_color)
            cr.set_line_width(self.stroke_width / scale)
            cr.stroke_preserve()
        cr.new_path()
    if InterpolatableProblem.UNDERWEIGHT in problem_types or InterpolatableProblem.OVERWEIGHT in problem_types:
        perContourPen = PerContourOrComponentPen(RecordingPen, glyphset=glyphset)
        recording.replay(perContourPen)
        for problem in problems:
            if problem['type'] in (InterpolatableProblem.UNDERWEIGHT, InterpolatableProblem.OVERWEIGHT):
                contour = perContourPen.value[problem['contour']]
                contour.replay(CairoPen(glyphset, cr))
                cr.set_source_rgba(*self.weight_issue_contour_color)
                cr.fill()
    if any((t in problem_types for t in {InterpolatableProblem.NOTHING, InterpolatableProblem.NODE_COUNT, InterpolatableProblem.NODE_INCOMPATIBILITY})):
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        for segment, args in decomposedRecording.value:
            if not args:
                continue
            x, y = args[-1]
            cr.move_to(x, y)
            cr.line_to(x, y)
        cr.set_source_rgba(*self.oncurve_node_color)
        cr.set_line_width(self.oncurve_node_diameter / scale)
        cr.stroke()
        for segment, args in decomposedRecording.value:
            if not args:
                continue
            for x, y in args[:-1]:
                cr.move_to(x, y)
                cr.line_to(x, y)
        cr.set_source_rgba(*self.offcurve_node_color)
        cr.set_line_width(self.offcurve_node_diameter / scale)
        cr.stroke()
        for segment, args in decomposedRecording.value:
            if not args:
                pass
            elif segment in ('moveTo', 'lineTo'):
                cr.move_to(*args[0])
            elif segment == 'qCurveTo':
                for x, y in args:
                    cr.line_to(x, y)
                cr.new_sub_path()
                cr.move_to(*args[-1])
            elif segment == 'curveTo':
                cr.line_to(*args[0])
                cr.new_sub_path()
                cr.move_to(*args[1])
                cr.line_to(*args[2])
                cr.new_sub_path()
                cr.move_to(*args[-1])
            else:
                continue
        cr.set_source_rgba(*self.handle_color)
        cr.set_line_width(self.handle_width / scale)
        cr.stroke()
    matching = None
    for problem in problems:
        if problem['type'] == InterpolatableProblem.CONTOUR_ORDER:
            matching = problem['value_2']
            colors = cycle(self.contour_colors)
            perContourPen = PerContourOrComponentPen(RecordingPen, glyphset=glyphset)
            recording.replay(perContourPen)
            for i, contour in enumerate(perContourPen.value):
                if matching[i] == i:
                    continue
                color = next(colors)
                contour.replay(CairoPen(glyphset, cr))
                cr.set_source_rgba(*color, self.contour_alpha)
                cr.fill()
    for problem in problems:
        if problem['type'] in (InterpolatableProblem.NOTHING, InterpolatableProblem.WRONG_START_POINT):
            idx = problem.get('contour')
            if idx is not None and which == 1 and ('value_2' in problem):
                perContourPen = PerContourOrComponentPen(RecordingPen, glyphset=glyphset)
                decomposedRecording.replay(perContourPen)
                points = SimpleRecordingPointPen()
                converter = SegmentToPointPen(points, False)
                perContourPen.value[idx if matching is None else matching[idx]].replay(converter)
                targetPoint = points.value[problem['value_2']][0]
                cr.save()
                cr.translate(*targetPoint)
                cr.scale(1 / scale, 1 / scale)
                self.draw_dot(cr, diameter=self.corrected_start_point_size, color=self.corrected_start_point_color)
                cr.restore()
            if which == 0 or not problem.get('reversed'):
                color = self.start_point_color
            else:
                color = self.wrong_start_point_color
            first_pt = None
            i = 0
            cr.save()
            for segment, args in decomposedRecording.value:
                if segment == 'moveTo':
                    first_pt = args[0]
                    continue
                if first_pt is None:
                    continue
                if segment == 'closePath':
                    second_pt = first_pt
                else:
                    second_pt = args[0]
                if idx is None or i == idx:
                    cr.save()
                    first_pt = complex(*first_pt)
                    second_pt = complex(*second_pt)
                    length = abs(second_pt - first_pt)
                    cr.translate(first_pt.real, first_pt.imag)
                    if length:
                        cr.rotate(math.atan2(second_pt.imag - first_pt.imag, second_pt.real - first_pt.real))
                        cr.scale(1 / scale, 1 / scale)
                        self.draw_arrow(cr, color=color)
                    else:
                        cr.scale(1 / scale, 1 / scale)
                        self.draw_dot(cr, diameter=self.corrected_start_point_size, color=color)
                    cr.restore()
                    if idx is not None:
                        break
                first_pt = None
                i += 1
            cr.restore()
        if problem['type'] == InterpolatableProblem.KINK:
            idx = problem.get('contour')
            perContourPen = PerContourOrComponentPen(RecordingPen, glyphset=glyphset)
            decomposedRecording.replay(perContourPen)
            points = SimpleRecordingPointPen()
            converter = SegmentToPointPen(points, False)
            perContourPen.value[idx if matching is None else matching[idx]].replay(converter)
            targetPoint = points.value[problem['value']][0]
            cr.save()
            cr.translate(*targetPoint)
            cr.scale(1 / scale, 1 / scale)
            if midway:
                self.draw_circle(cr, diameter=self.kink_circle_size, stroke_width=self.kink_circle_stroke_width, color=self.kink_circle_color)
            else:
                self.draw_dot(cr, diameter=self.kink_point_size, color=self.kink_point_color)
            cr.restore()
    return scale