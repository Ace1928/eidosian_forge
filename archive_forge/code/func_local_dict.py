import re
import threading
def local_dict():
    global config_local, local
    try:
        return config_local.wsgi_dict
    except NameError:
        config_local = threading.local()
        config_local.wsgi_dict = result = {}
        return result
    except AttributeError:
        config_local.wsgi_dict = result = {}
        return result