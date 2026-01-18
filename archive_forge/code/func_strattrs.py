@staticmethod
def strattrs(attrs):
    return ''.join((' %s="%s"' % (n, v.replace('"', '&quot;')) for n, v in attrs))