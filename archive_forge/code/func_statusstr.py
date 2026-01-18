def statusstr(status):
    if status in statusmap:
        return statusmap[status]
    else:
        return repr(status)