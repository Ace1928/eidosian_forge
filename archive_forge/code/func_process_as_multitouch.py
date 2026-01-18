import os
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect
def process_as_multitouch(tv_sec, tv_usec, ev_type, ev_code, ev_value):
    if ev_type == EV_SYN:
        if ev_code == SYN_MT_REPORT:
            if 'id' not in point:
                return
            l_points.append(point.copy())
        elif ev_code == SYN_REPORT:
            process(l_points)
            del l_points[:]
    elif ev_type == EV_MSC and ev_code in (MSC_RAW, MSC_SCAN):
        pass
    elif ev_code == ABS_MT_TRACKING_ID:
        point.clear()
        point['id'] = ev_value
    elif ev_code == ABS_MT_POSITION_X:
        val = normalize(ev_value, range_min_position_x, range_max_position_x)
        assign_coord(point, val, invert_x, 'xy')
    elif ev_code == ABS_MT_POSITION_Y:
        val = 1.0 - normalize(ev_value, range_min_position_y, range_max_position_y)
        assign_coord(point, val, invert_y, 'yx')
    elif ev_code == ABS_MT_ORIENTATION:
        point['orientation'] = ev_value
    elif ev_code == ABS_MT_BLOB_ID:
        point['blobid'] = ev_value
    elif ev_code == ABS_MT_PRESSURE:
        point['pressure'] = normalize(ev_value, range_min_pressure, range_max_pressure)
    elif ev_code == ABS_MT_TOUCH_MAJOR:
        point['size_w'] = ev_value
    elif ev_code == ABS_MT_TOUCH_MINOR:
        point['size_h'] = ev_value