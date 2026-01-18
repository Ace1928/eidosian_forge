from paste.httpexceptions import HTTPUnauthorized
from paste.httpheaders import (
import time, random
from urllib.parse import quote as url_quote
def make_digest(app, global_conf, realm, authfunc, **kw):
    """
    Grant access via digest authentication

    Config looks like this::

      [filter:grant]
      use = egg:Paste#auth_digest
      realm=myrealm
      authfunc=somepackage.somemodule:somefunction

    """
    from paste.util.import_string import eval_import
    import types
    authfunc = eval_import(authfunc)
    assert isinstance(authfunc, types.FunctionType), 'authfunc must resolve to a function'
    return AuthDigestHandler(app, realm, authfunc)