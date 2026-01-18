from paste.request import construct_url, parse_formvars
def make_form(app, global_conf, realm, authfunc, **kw):
    """
    Grant access via form authentication

    Config looks like this::

      [filter:grant]
      use = egg:Paste#auth_form
      realm=myrealm
      authfunc=somepackage.somemodule:somefunction

    """
    from paste.util.import_string import eval_import
    import types
    authfunc = eval_import(authfunc)
    assert isinstance(authfunc, types.FunctionType), 'authfunc must resolve to a function'
    template = kw.get('template')
    if template is not None:
        template = eval_import(template)
        assert isinstance(template, str), 'template must resolve to a string'
    return AuthFormHandler(app, authfunc, template)