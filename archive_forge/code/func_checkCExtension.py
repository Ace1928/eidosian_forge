def checkCExtension(*, warn, warn_c=False):
    import frozendict as cool
    res = cool.c_ext
    if warn and res == warn_c:
        if warn_c:
            msg = 'C Extension version, monkeypatch will be not applied'
        else:
            msg = 'Pure Python version, monkeypatch will be not applied'
        import warnings
        warnings.warn(msg, MonkeypatchWarning)
    return res