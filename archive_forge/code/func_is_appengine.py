import os
def is_appengine():
    return is_local_appengine() or is_prod_appengine()