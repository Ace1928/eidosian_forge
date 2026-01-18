from nltk.internals import Counter
from nltk.sem.logic import APP, LogicParser
def compile_pos(self, index_counter, glueFormulaFactory):
    """
        From Iddo Lev's PhD Dissertation p108-109

        :param index_counter: ``Counter`` for unique indices
        :param glueFormulaFactory: ``GlueFormula`` for creating new glue formulas
        :return: (``Expression``,set) for the compiled linear logic and any newly created glue formulas
        """
    a, a_new = self.antecedent.compile_neg(index_counter, glueFormulaFactory)
    c, c_new = self.consequent.compile_pos(index_counter, glueFormulaFactory)
    return (ImpExpression(a, c), a_new + c_new)