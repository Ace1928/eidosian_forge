from emoji.unicode_codes.data_dict import *
def get_emoji_unicode_dict(lang):
    """Generate dict containing all fully-qualified and component emoji name for a language
    The dict is only generated once per language and then cached in _EMOJI_UNICODE[lang]"""
    if _EMOJI_UNICODE[lang] is None:
        _EMOJI_UNICODE[lang] = {data[lang]: emj for emj, data in EMOJI_DATA.items() if lang in data and data['status'] <= STATUS['fully_qualified']}
    return _EMOJI_UNICODE[lang]