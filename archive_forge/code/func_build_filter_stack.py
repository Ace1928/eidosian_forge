from sqlparse import filters
from sqlparse.exceptions import SQLParseError
def build_filter_stack(stack, options):
    """Setup and return a filter stack.

    Args:
      stack: :class:`~sqlparse.filters.FilterStack` instance
      options: Dictionary with options validated by validate_options.
    """
    if options.get('keyword_case'):
        stack.preprocess.append(filters.KeywordCaseFilter(options['keyword_case']))
    if options.get('identifier_case'):
        stack.preprocess.append(filters.IdentifierCaseFilter(options['identifier_case']))
    if options.get('truncate_strings'):
        stack.preprocess.append(filters.TruncateStringFilter(width=options['truncate_strings'], char=options['truncate_char']))
    if options.get('use_space_around_operators', False):
        stack.enable_grouping()
        stack.stmtprocess.append(filters.SpacesAroundOperatorsFilter())
    if options.get('strip_comments'):
        stack.enable_grouping()
        stack.stmtprocess.append(filters.StripCommentsFilter())
    if options.get('strip_whitespace') or options.get('reindent'):
        stack.enable_grouping()
        stack.stmtprocess.append(filters.StripWhitespaceFilter())
    if options.get('reindent'):
        stack.enable_grouping()
        stack.stmtprocess.append(filters.ReindentFilter(char=options['indent_char'], width=options['indent_width'], indent_after_first=options['indent_after_first'], indent_columns=options['indent_columns'], wrap_after=options['wrap_after'], comma_first=options['comma_first']))
    if options.get('reindent_aligned', False):
        stack.enable_grouping()
        stack.stmtprocess.append(filters.AlignedIndentFilter(char=options['indent_char']))
    if options.get('right_margin'):
        stack.enable_grouping()
        stack.stmtprocess.append(filters.RightMarginFilter(width=options['right_margin']))
    if options.get('output_format'):
        frmt = options['output_format']
        if frmt.lower() == 'php':
            fltr = filters.OutputPHPFilter()
        elif frmt.lower() == 'python':
            fltr = filters.OutputPythonFilter()
        else:
            fltr = None
        if fltr is not None:
            stack.postprocess.append(fltr)
    return stack