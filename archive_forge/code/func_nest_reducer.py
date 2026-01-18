from pythran.analyses import ImportedIds
from pythran.passmanager import Transformation
from pythran.conversion import mangle
import pythran.metadata as metadata
import gast as ast
from functools import reduce
@staticmethod
def nest_reducer(x, g):
    """
        Create a ast.For node from a comprehension and another node.

        g is an ast.comprehension.
        x is the code that have to be executed.

        Examples
        --------
        >> [i for i in range(2)]

        Becomes

        >> for i in range(2):
        >>    ... x code with if clauses ...

        It is a reducer as it can be call recursively for mutli generator.

        Ex : >> [i, j for i in range(2) for j in range(4)]
        """

    def wrap_in_ifs(node, ifs):
        """
            Wrap comprehension content in all possibles if clauses.

            Examples
            --------
            >> [i for i in range(2) if i < 3 if 0 < i]

            Becomes

            >> for i in range(2):
            >>    if i < 3:
            >>        if 0 < i:
            >>            ... the code from `node` ...

            Note the nested ifs clauses.
            """
        return reduce(lambda n, if_: ast.If(if_, [n], []), ifs, node)
    return ast.For(g.target, g.iter, [wrap_in_ifs(x, g.ifs)], [], None)