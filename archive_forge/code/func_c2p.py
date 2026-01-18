from sympy.utilities.pkgdata import get_resource
from sympy.utilities.decorator import doctest_depends_on
@doctest_depends_on(modules=('lxml',))
def c2p(mml, simple=False):
    """Transforms a document in MathML content (like the one that sympy produces)
    in one document in MathML presentation, more suitable for printing, and more
    widely accepted

    Examples
    ========

    >>> from sympy.utilities.mathml import c2p
    >>> mml = '<apply> <exp/> <cn>2</cn> </apply>'
    >>> c2p(mml,simple=True) != c2p(mml,simple=False)
    True

    """
    if not mml.startswith('<math'):
        mml = add_mathml_headers(mml)
    if simple:
        return apply_xsl(mml, 'mathml/data/simple_mmlctop.xsl')
    return apply_xsl(mml, 'mathml/data/mmlctop.xsl')