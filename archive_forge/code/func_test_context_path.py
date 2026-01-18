import array
import base64
import contextlib
import gc
import io
import math
import os
import shutil
import sys
import tempfile
import cairocffi
import pikepdf
import pytest
from . import (
def test_context_path():
    surface = ImageSurface(cairocffi.FORMAT_ARGB32, 1, 1)
    context = Context(surface)
    assert context.copy_path() == []
    assert context.has_current_point() is False
    assert context.get_current_point() == (0, 0)
    context.arc(100, 200, 20, math.pi / 2, 0)
    path_1 = context.copy_path()
    assert path_1[0] == (cairocffi.PATH_MOVE_TO, (100, 220))
    assert len(path_1) > 1
    assert all((part[0] == cairocffi.PATH_CURVE_TO for part in path_1[1:]))
    assert context.has_current_point() is True
    assert context.get_current_point() == (120, 200)
    context.new_sub_path()
    assert context.copy_path() == path_1
    assert context.has_current_point() is False
    assert context.get_current_point() == (0, 0)
    context.new_path()
    assert context.copy_path() == []
    assert context.has_current_point() is False
    assert context.get_current_point() == (0, 0)
    context.arc_negative(100, 200, 20, math.pi / 2, 0)
    path_2 = context.copy_path()
    assert path_2[0] == (cairocffi.PATH_MOVE_TO, (100, 220))
    assert len(path_2) > 1
    assert all((part[0] == cairocffi.PATH_CURVE_TO for part in path_2[1:]))
    assert path_2 != path_1
    context.new_path()
    context.rectangle(10, 20, 100, 200)
    path = context.copy_path()
    if path[-1] == (cairocffi.PATH_MOVE_TO, (10, 20)):
        path = path[:-1]
    assert path == [(cairocffi.PATH_MOVE_TO, (10, 20)), (cairocffi.PATH_LINE_TO, (110, 20)), (cairocffi.PATH_LINE_TO, (110, 220)), (cairocffi.PATH_LINE_TO, (10, 220)), (cairocffi.PATH_CLOSE_PATH, ())]
    assert context.path_extents() == (10, 20, 110, 220)
    context.new_path()
    context.move_to(10, 20)
    context.line_to(10, 30)
    context.rel_move_to(2, 5)
    context.rel_line_to(2, 5)
    context.curve_to(20, 30, 70, 50, 100, 120)
    context.rel_curve_to(20, 30, 70, 50, 100, 120)
    context.close_path()
    path = context.copy_path()
    if path[-1] == (cairocffi.PATH_MOVE_TO, (12, 35)):
        path = path[:-1]
    assert path == [(cairocffi.PATH_MOVE_TO, (10, 20)), (cairocffi.PATH_LINE_TO, (10, 30)), (cairocffi.PATH_MOVE_TO, (12, 35)), (cairocffi.PATH_LINE_TO, (14, 40)), (cairocffi.PATH_CURVE_TO, (20, 30, 70, 50, 100, 120)), (cairocffi.PATH_CURVE_TO, (120, 150, 170, 170, 200, 240)), (cairocffi.PATH_CLOSE_PATH, ())]
    context.new_path()
    context.move_to(10, 15)
    context.curve_to(20, 30, 70, 50, 100, 120)
    assert context.copy_path() == [(cairocffi.PATH_MOVE_TO, (10, 15)), (cairocffi.PATH_CURVE_TO, (20, 30, 70, 50, 100, 120))]
    path = context.copy_path_flat()
    assert len(path) > 2
    assert path[0] == (cairocffi.PATH_MOVE_TO, (10, 15))
    assert all((part[0] == cairocffi.PATH_LINE_TO for part in path[1:]))
    assert path[-1] == (cairocffi.PATH_LINE_TO, (100, 120))
    context.new_path()
    context.move_to(10, 20)
    context.line_to(10, 30)
    path = context.copy_path()
    assert path == [(cairocffi.PATH_MOVE_TO, (10, 20)), (cairocffi.PATH_LINE_TO, (10, 30))]
    additional_path = [(cairocffi.PATH_LINE_TO, (30, 150))]
    context.append_path(additional_path)
    assert context.copy_path() == path + additional_path
    with pytest.raises(ValueError):
        context.append_path([(cairocffi.PATH_LINE_TO, (30, 150, 1))])
    with pytest.raises(ValueError):
        context.append_path([(cairocffi.PATH_LINE_TO, (30, 150, 1, 4))])