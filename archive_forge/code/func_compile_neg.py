from nltk.internals import Counter
from nltk.sem.logic import APP, LogicParser
def compile_neg(self, index_counter, glueFormulaFactory):
    """
        From Iddo Lev's PhD Dissertation p108-109

        :param index_counter: ``Counter`` for unique indices
        :param glueFormulaFactory: ``GlueFormula`` for creating new glue formulas
        :return: (``Expression``,list of ``GlueFormula``) for the compiled linear logic and any newly created glue formulas
        """
    a, a_new = self.antecedent.compile_pos(index_counter, glueFormulaFactory)
    c, c_new = self.consequent.compile_neg(index_counter, glueFormulaFactory)
    fresh_index = index_counter.get()
    c.dependencies.append(fresh_index)
    new_v = glueFormulaFactory('v%s' % fresh_index, a, {fresh_index})
    return (c, a_new + c_new + [new_v])