from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.assumptions.ask_generated import get_all_known_facts
from sympy.assumptions.assume import global_assumptions, AppliedPredicate
from sympy.assumptions.sathandlers import class_fact_registry
from sympy.core import oo
from sympy.logic.inference import satisfiable
from sympy.assumptions.cnf import CNF, EncodedCNF
def get_all_relevant_facts(proposition, assumptions, context, use_known_facts=True, iterations=oo):
    """
    Extract all relevant facts from *proposition* and *assumptions*.

    This function extracts the facts by recursively calling
    ``get_relevant_clsfacts()``. Extracted facts are converted to
    ``EncodedCNF`` and returned.

    Parameters
    ==========

    proposition : sympy.assumptions.cnf.CNF
        CNF generated from proposition expression.

    assumptions : sympy.assumptions.cnf.CNF
        CNF generated from assumption expression.

    context : sympy.assumptions.cnf.CNF
        CNF generated from assumptions context.

    use_known_facts : bool, optional.
        If ``True``, facts from ``sympy.assumptions.ask_generated``
        module are encoded as well.

    iterations : int, optional.
        Number of times that relevant facts are recursively extracted.
        Default is infinite times until no new fact is found.

    Returns
    =======

    sympy.assumptions.cnf.EncodedCNF

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.cnf import CNF
    >>> from sympy.assumptions.satask import get_all_relevant_facts
    >>> from sympy.abc import x, y
    >>> props = CNF.from_prop(Q.nonzero(x*y))
    >>> assump = CNF.from_prop(Q.nonzero(x))
    >>> context = CNF.from_prop(Q.nonzero(y))
    >>> get_all_relevant_facts(props, assump, context) #doctest: +SKIP
    <sympy.assumptions.cnf.EncodedCNF at 0x7f09faa6ccd0>

    """
    i = 0
    relevant_facts = CNF()
    all_exprs = set()
    while True:
        if i == 0:
            exprs = extract_predargs(proposition, assumptions, context)
        all_exprs |= exprs
        exprs, relevant_facts = get_relevant_clsfacts(exprs, relevant_facts)
        i += 1
        if i >= iterations:
            break
        if not exprs:
            break
    if use_known_facts:
        known_facts_CNF = CNF()
        known_facts_CNF.add_clauses(get_all_known_facts())
        kf_encoded = EncodedCNF()
        kf_encoded.from_cnf(known_facts_CNF)

        def translate_literal(lit, delta):
            if lit > 0:
                return lit + delta
            else:
                return lit - delta

        def translate_data(data, delta):
            return [{translate_literal(i, delta) for i in clause} for clause in data]
        data = []
        symbols = []
        n_lit = len(kf_encoded.symbols)
        for i, expr in enumerate(all_exprs):
            symbols += [pred(expr) for pred in kf_encoded.symbols]
            data += translate_data(kf_encoded.data, i * n_lit)
        encoding = dict(list(zip(symbols, range(1, len(symbols) + 1))))
        ctx = EncodedCNF(data, encoding)
    else:
        ctx = EncodedCNF()
    ctx.add_from_cnf(relevant_facts)
    return ctx