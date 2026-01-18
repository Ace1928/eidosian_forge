from pyparsing import (
import pydot
def parse_dot_data(s):
    """Parse DOT description in (unicode) string `s`.

    @return: Graphs that result from parsing.
    @rtype: `list` of `pydot.Dot`
    """
    global top_graphs
    top_graphs = list()
    try:
        graphparser = graph_definition()
        graphparser.parseWithTabs()
        tokens = graphparser.parseString(s)
        return list(tokens)
    except ParseException as err:
        print(err.line)
        print(' ' * (err.column - 1) + '^')
        print(err)
        return None