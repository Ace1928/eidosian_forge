from pyparsing import col,Word,Optional,alphas,nums
def tableValue(expr, colstart, colend):
    empty_cell_is_zero = False
    if empty_cell_is_zero:
        return Optional(expr.copy().addCondition(mustMatchCols(colstart, colend), message='text not in expected columns'), default=0)
    else:
        return Optional(expr.copy().addCondition(mustMatchCols(colstart, colend), message='text not in expected columns'))