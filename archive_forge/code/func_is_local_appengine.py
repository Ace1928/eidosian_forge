import os
def is_local_appengine():
    return 'APPENGINE_RUNTIME' in os.environ and os.environ.get('SERVER_SOFTWARE', '').startswith('Development/')