import re
def port_range(start, end, proto, randomly_available_port=False):
    if not start:
        return start
    if not end:
        return [start + proto]
    if randomly_available_port:
        return [f'{start}-{end}' + proto]
    return [str(port) + proto for port in range(int(start), int(end) + 1)]