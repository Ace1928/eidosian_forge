import argparse
from osc_lib.i18n import _
def get_tag_filtering_args(parsed_args, args):
    """Adds the tag arguments to an args list.

    Intended to be used to append the tags to an argument list that will be
    used for service client.

    :param parsed_args: Parsed argument object returned by argparse parse_args.
    :param args: The argument list to add tags to.
    """
    if parsed_args.tags:
        args['tags'] = ','.join(parsed_args.tags)
    if parsed_args.any_tags:
        args['any_tags'] = ','.join(parsed_args.any_tags)
    if parsed_args.not_tags:
        args['not_tags'] = ','.join(parsed_args.not_tags)
    if parsed_args.not_any_tags:
        args['not_any_tags'] = ','.join(parsed_args.not_any_tags)