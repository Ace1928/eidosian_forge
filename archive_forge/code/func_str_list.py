from io import StringIO
def str_list(val):
    with StringIO() as buf:
        buf.write('[')
        first = True
        for item in val:
            if not first:
                buf.write(', ')
            buf.write(str(item))
            first = False
        buf.write(']')
        return buf.getvalue()