def ro(C, strict=None, base_mros=None, log_changed_ro=None, use_legacy_ro=None):
    """
    ro(C) -> list

    Compute the precedence list (mro) according to C3.

    :return: A fresh `list` object.

    .. versionchanged:: 5.0.0
       Add the *strict*, *log_changed_ro* and *use_legacy_ro*
       keyword arguments. These are provisional and likely to be
       removed in the future. They are most useful for testing.
    """
    resolver = C3.resolver(C, strict, base_mros)
    mro = resolver.mro()
    log_changed = log_changed_ro if log_changed_ro is not None else resolver.LOG_CHANGED_IRO
    use_legacy = use_legacy_ro if use_legacy_ro is not None else resolver.USE_LEGACY_IRO
    if log_changed or use_legacy:
        legacy_ro = resolver.legacy_ro
        assert isinstance(legacy_ro, list)
        assert isinstance(mro, list)
        changed = legacy_ro != mro
        if changed:
            legacy_without_root = [x for x in legacy_ro if x is not _ROOT]
            mro_without_root = [x for x in mro if x is not _ROOT]
            changed = legacy_without_root != mro_without_root
        if changed:
            comparison = _ROComparison(resolver, mro, legacy_ro)
            _logger().warning('Object %r has different legacy and C3 MROs:\n%s', C, comparison)
        if resolver.had_inconsistency and legacy_ro == mro:
            comparison = _ROComparison(resolver, mro, legacy_ro)
            _logger().warning('Object %r had inconsistent IRO and used the legacy RO:\n%s\nInconsistency entered at:\n%s', C, comparison, resolver.direct_inconsistency)
        if use_legacy:
            return legacy_ro
    return mro