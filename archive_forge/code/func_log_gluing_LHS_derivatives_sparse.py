from snappy import snap
from snappy.sage_helper import _within_sage, sage_method
def log_gluing_LHS_derivatives_sparse(self, shapes):
    """
        A column-sparse matrix version of log_gluing_LHS_derivatives_sparse.
        The result is a list of list of pairs. Each list of pairs corresponds
        to a column, a pair being (index of row, value) where the index is
        increasing.
        """
    BaseField = shapes[0].parent()
    zero = BaseField(0)
    one = BaseField(1)
    gluing_LHS_derivatives = []
    for eqns_column, shape in zip(self.sparse_equations, shapes):
        shape_inverse = one / shape
        one_minus_shape_inverse = one / (one - shape)
        column = []
        for r, (a, b) in eqns_column:
            derivative = zero
            if not a == 0:
                derivative = BaseField(int(a)) * shape_inverse
            if not b == 0:
                derivative -= BaseField(int(b)) * one_minus_shape_inverse
            column.append((r, derivative))
        gluing_LHS_derivatives.append(column)
    return gluing_LHS_derivatives