import os
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect
def process_as_mouse_or_keyboard(tv_sec, tv_usec, ev_type, ev_code, ev_value):
    if ev_type == EV_SYN:
        if ev_code == SYN_REPORT:
            process([point])
            if 'button' in point and point['button'].startswith('scroll'):
                del point['button']
                point['id'] += 1
                point['_avoid'] = True
                process([point])
    elif ev_type == EV_REL:
        if ev_code == 0:
            assign_rel_coord(point, min(1.0, max(-1.0, ev_value / 1000.0)), invert_x, 'xy')
        elif ev_code == 1:
            assign_rel_coord(point, min(1.0, max(-1.0, ev_value / 1000.0)), invert_y, 'yx')
        elif ev_code == 8:
            b = 'scrollup' if ev_value < 0 else 'scrolldown'
            if 'button' not in point:
                point['button'] = b
                point['id'] += 1
                if '_avoid' in point:
                    del point['_avoid']
    elif ev_type != EV_KEY:
        if ev_code == ABS_X:
            val = normalize(ev_value, range_min_abs_x, range_max_abs_x)
            assign_coord(point, val, invert_x, 'xy')
        elif ev_code == ABS_Y:
            val = 1.0 - normalize(ev_value, range_min_abs_y, range_max_abs_y)
            assign_coord(point, val, invert_y, 'yx')
        elif ev_code == ABS_PRESSURE:
            point['pressure'] = normalize(ev_value, range_min_abs_pressure, range_max_abs_pressure)
    else:
        buttons = {272: 'left', 273: 'right', 274: 'middle', 275: 'side', 276: 'extra', 277: 'forward', 278: 'back', 279: 'task', 330: 'touch', 320: 'pen'}
        if ev_code in buttons.keys():
            if ev_value:
                if 'button' not in point:
                    point['button'] = buttons[ev_code]
                    point['id'] += 1
                    if '_avoid' in point:
                        del point['_avoid']
            elif 'button' in point:
                if point['button'] == buttons[ev_code]:
                    del point['button']
                    point['id'] += 1
                    point['_avoid'] = True
        else:
            if not 0 <= ev_value <= 1:
                return
            if ev_code not in keyboard_keys:
                Logger.warn('HIDInput: unhandled HID code: {}'.format(ev_code))
                return
            z = keyboard_keys[ev_code][-1 if 'shift' in Window._modifiers else 0]
            if z.lower() not in Keyboard.keycodes:
                Logger.warn('HIDInput: unhandled character: {}'.format(z))
                return
            keycode = Keyboard.keycodes[z.lower()]
            if ev_value == 1:
                if z == 'shift' or z == 'alt':
                    Window._modifiers.append(z)
                elif z.endswith('ctrl'):
                    Window._modifiers.append('ctrl')
                dispatch_queue.append(('key_down', (keycode, ev_code, keys_str.get(z, z), Window._modifiers)))
            elif ev_value == 0:
                dispatch_queue.append(('key_up', (keycode, ev_code, keys_str.get(z, z), Window._modifiers)))
                if (z == 'shift' or z == 'alt') and z in Window._modifiers:
                    Window._modifiers.remove(z)
                elif z.endswith('ctrl') and 'ctrl' in Window._modifiers:
                    Window._modifiers.remove('ctrl')