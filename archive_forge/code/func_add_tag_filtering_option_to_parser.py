import argparse
from osc_lib.i18n import _
def add_tag_filtering_option_to_parser(parser, resource_name, enhance_help=lambda _h: _h):
    """Add tag filtering options to a parser.

    :param parser: argparse.Argument parser object.
    :param resource_name: Description of the object being filtered.
    :param enhance_help: A callable accepting a single parameter, the
        (translated) help string, and returning a (translated) help string. May
        be used by a caller wishing to add qualifying text, such as "Applies to
        version XYZ only", to the help strings for all options produced by this
        method.
    """
    parser.add_argument('--tags', metavar='<tag>[,<tag>,...]', action=_CommaListAction, help=enhance_help(_('List %s which have all given tag(s) (Comma-separated list of tags)') % resource_name))
    parser.add_argument('--any-tags', metavar='<tag>[,<tag>,...]', action=_CommaListAction, help=enhance_help(_('List %s which have any given tag(s) (Comma-separated list of tags)') % resource_name))
    parser.add_argument('--not-tags', metavar='<tag>[,<tag>,...]', action=_CommaListAction, help=enhance_help(_('Exclude %s which have all given tag(s) (Comma-separated list of tags)') % resource_name))
    parser.add_argument('--not-any-tags', metavar='<tag>[,<tag>,...]', action=_CommaListAction, help=enhance_help(_('Exclude %s which have any given tag(s) (Comma-separated list of tags)') % resource_name))