import re
from urllib.parse import quote
from bleach import callbacks as linkify_callbacks
from bleach import html5lib_shim
def build_email_re(tlds=TLDS):
    """Builds the email regex used by linkifier

    If you want a different set of tlds, pass those in and stomp on the existing ``email_re``::

        from bleach import linkifier

        my_email_re = linkifier.build_email_re(my_tlds_list)

        linker = LinkifyFilter(email_re=my_url_re)

    """
    return re.compile('(?<!//)\n        (([-!#$%&\'*+/=?^_`{{}}|~0-9A-Z]+\n            (\\.[-!#$%&\'*+/=?^_`{{}}|~0-9A-Z]+)*  # dot-atom\n        |^"([\\001-\\010\\013\\014\\016-\\037!#-\\[\\]-\\177]\n            |\\\\[\\001-\\011\\013\\014\\016-\\177])*"  # quoted-string\n        )@(?:[A-Z0-9](?:[A-Z0-9-]{{0,61}}[A-Z0-9])?\\.)+(?:{0}))  # domain\n        '.format('|'.join(tlds)), re.IGNORECASE | re.MULTILINE | re.VERBOSE)