from .schema import rest_translation
def inv_translate_uri(uri):
    inv_table = dict(zip(rest_translation.values(), rest_translation.keys()))
    for key in inv_table.keys():
        uri = uri.replace('/%s' % key, '/%s' % inv_table[key])
    return uri