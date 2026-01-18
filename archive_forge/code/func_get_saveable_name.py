def get_saveable_name(cls_or_obj):
    return getattr(cls_or_obj, _LEGACY_SAVEABLE_NAME, None)