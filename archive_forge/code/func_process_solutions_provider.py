from .component import NonZeroDimensionalComponent
from . import processFileBase
from . import processRurFile
from . import utilities
from . import coordinates
from .polynomial import Polynomial
from .ptolemyVarietyPrimeIdealGroebnerBasis import PtolemyVarietyPrimeIdealGroebnerBasis
def process_solutions_provider(py_eval, manifold_thunk, text, for_dimension, variables):
    rur_section = processFileBase.find_section(text, 'MAPLE=LIKE=RUR')
    if rur_section:
        assert len(rur_section) == 1
        return SolutionContainer(coordinates.PtolemyCoordinates(processRurFile.parse_maple_like_rur(rur_section[0]), is_numerical=False, py_eval_section=py_eval, manifold_thunk=manifold_thunk))
    gb_section = processFileBase.find_section(text, 'GROEBNER=BASIS')
    if gb_section:
        assert len(gb_section) == 1
        params, body = processFileBase.extract_parameters_and_body_from_section(gb_section[0])
        body = processFileBase.remove_optional_outer_square_brackets(utilities.join_long_lines_deleting_whitespace(body))
        polys = [Polynomial.parse_string(p) for p in body.replace('\n', ' ').split(',')]
        if 'TERM=ORDER' not in params.keys():
            raise Exception('No term order given for Groebner basis')
        term_order = params['TERM=ORDER'].strip().lower()
        return PtolemyVarietyPrimeIdealGroebnerBasis(polys=polys, term_order=term_order, size=None, dimension=0, is_prime=True, free_variables=None, py_eval=py_eval, manifold_thunk=manifold_thunk)
    rs_rur = processFileBase.find_section(text, 'RS=RUR')
    if rs_rur:
        assert len(rs_rur) == 1
        return SolutionContainer(coordinates.PtolemyCoordinates(processRurFile.parse_rs_rur(rs_rur[0], variables), is_numerical=False, py_eval_section=py_eval, manifold_thunk=manifold_thunk))
    raise Exception('No parsable solution type given: %s...' % text[:100])