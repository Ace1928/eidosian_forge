from __future__ import unicode_literals
from pybtex.style import FormattedEntry, FormattedBibliography
from pybtex.style.template import node, join
from pybtex.richtext import Symbol
from pybtex.plugin import Plugin, find_plugin
def format_entries(self, entries, bib_data=None):
    sorted_entries = self.sort(entries)
    labels = self.format_labels(sorted_entries)
    for label, entry in zip(labels, sorted_entries):
        yield self.format_entry(label, entry, bib_data=bib_data)