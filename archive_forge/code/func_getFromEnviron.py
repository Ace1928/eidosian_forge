import os, base64
@staticmethod
def getFromEnviron():
    url = None
    for key in ('http_proxy', 'https_proxy'):
        url = os.environ.get(key)
        if url:
            break
    if not url:
        return None
    dat = urlparse(url)
    port = 80 if dat.scheme == 'http' else 443
    if dat.port != None:
        port = int(dat.port)
    host = dat.hostname
    return HttpProxy((host, port), dat.username, dat.password)