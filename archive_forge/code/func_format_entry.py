from __future__ import unicode_literals
from pybtex.style import FormattedEntry, FormattedBibliography
from pybtex.style.template import node, join
from pybtex.richtext import Symbol
from pybtex.plugin import Plugin, find_plugin
def format_entry(self, label, entry, bib_data=None):
    context = {'entry': entry, 'style': self, 'bib_data': bib_data}
    try:
        get_template = getattr(self, 'get_{}_template'.format(entry.type))
    except AttributeError:
        format_method = getattr(self, 'format_' + entry.type)
        text = format_method(context)
    else:
        text = get_template(entry).format_data(context)
    return FormattedEntry(entry.key, text, label)