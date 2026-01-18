import collections.abc
import itertools
import linecache
import sys
import textwrap
from contextlib import suppress
def format_frame_summary(self, frame_summary):
    """Format the lines for a single FrameSummary.

        Returns a string representing one frame involved in the stack. This
        gets called for every frame to be printed in the stack summary.
        """
    row = []
    row.append('  File "{}", line {}, in {}\n'.format(frame_summary.filename, frame_summary.lineno, frame_summary.name))
    if frame_summary.line:
        stripped_line = frame_summary.line.strip()
        row.append('    {}\n'.format(stripped_line))
        line = frame_summary._original_line
        orig_line_len = len(line)
        frame_line_len = len(frame_summary.line.lstrip())
        stripped_characters = orig_line_len - frame_line_len
        if frame_summary.colno is not None and frame_summary.end_colno is not None:
            start_offset = _byte_offset_to_character_offset(line, frame_summary.colno)
            end_offset = _byte_offset_to_character_offset(line, frame_summary.end_colno)
            code_segment = line[start_offset:end_offset]
            anchors = None
            if frame_summary.lineno == frame_summary.end_lineno:
                with suppress(Exception):
                    anchors = _extract_caret_anchors_from_line_segment(code_segment)
            else:
                end_offset = len(line.rstrip())
            if end_offset - start_offset < len(stripped_line) or (anchors and anchors.right_start_offset - anchors.left_end_offset > 0):
                dp_start_offset = _display_width(line, start_offset) + 1
                dp_end_offset = _display_width(line, end_offset) + 1
                row.append('    ')
                row.append(' ' * (dp_start_offset - stripped_characters))
                if anchors:
                    dp_left_end_offset = _display_width(code_segment, anchors.left_end_offset)
                    dp_right_start_offset = _display_width(code_segment, anchors.right_start_offset)
                    row.append(anchors.primary_char * dp_left_end_offset)
                    row.append(anchors.secondary_char * (dp_right_start_offset - dp_left_end_offset))
                    row.append(anchors.primary_char * (dp_end_offset - dp_start_offset - dp_right_start_offset))
                else:
                    row.append('^' * (dp_end_offset - dp_start_offset))
                row.append('\n')
    if frame_summary.locals:
        for name, value in sorted(frame_summary.locals.items()):
            row.append('    {name} = {value}\n'.format(name=name, value=value))
    return ''.join(row)