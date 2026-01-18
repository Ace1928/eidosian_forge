import reportlab
def hex32(i):
    return '0X%8.8X' % (int(i) & 4294967295)