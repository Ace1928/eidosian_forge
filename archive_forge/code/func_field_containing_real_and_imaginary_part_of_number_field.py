from ..sage_helper import _within_sage, sage_method, SageNotAvailable
@sage_method
def field_containing_real_and_imaginary_part_of_number_field(number_field):
    """
    Given a Sage number field number_field with a complex embedding z, return
               (real_number_field, real_part, imag_part).

    The number field real_number_field is the smallest number field containing
    the real part and imaginary part of every element in number_field.

    real_part and imag_part are elements in real_number_field which comes with
    a real embedding such that under this embedding, we have
    ``z = real_part + imag_part * I``.  ::

        sage: CF = ComplexField()
        sage: x = var('x')
        sage: nf = NumberField(x**2 + 1, 'x', embedding = CF(1.0j))
        sage: field_containing_real_and_imaginary_part_of_number_field(nf)
        (Number Field in x with defining polynomial x with x = 0, 0, 1)

        sage: nf = NumberField(x**2 + 7, 'x', embedding = CF(2.64575j))
        sage: field_containing_real_and_imaginary_part_of_number_field(nf)
        (Number Field in x with defining polynomial x^2 - 7 with x = 2.645751311064591?, 0, x)

        sage: nf = NumberField(x**3 + x**2 + 23, 'x', embedding = CF(1.1096 + 2.4317j))
        sage: field_containing_real_and_imaginary_part_of_number_field(nf)
        (Number Field in x with defining polynomial x^6 + 2*x^5 + 2*x^4 - 113/2*x^3 - 229/4*x^2 - 115/4*x - 575/8 with x = 3.541338405550421?, -20/14377*x^5 + 382/14377*x^4 + 526/14377*x^3 + 1533/14377*x^2 - 18262/14377*x - 10902/14377, 20/14377*x^5 - 382/14377*x^4 - 526/14377*x^3 - 1533/14377*x^2 + 32639/14377*x + 10902/14377)
    """
    equations = [_real_or_imaginary_part_for_polynomial_in_complex_variable(number_field.defining_polynomial(), start) for start in [0, 1]]
    k = 0
    extra_prec = 0
    CIF = ComplexIntervalField()
    z_val = CIF(number_field.gen_embedding())
    x_val = z_val.real()
    y_val = z_val.imag()
    while k < 100 and extra_prec < 5:
        xprime_val = x_val + k * y_val
        equations_for_xprime = [eqn.substitute(x=var('x') - k * var('y')) for eqn in equations]
        try:
            solution = _solve_two_equations(equations_for_xprime[0], equations_for_xprime[1], xprime_val, y_val)
            if solution:
                real_number_field, y_expression = solution
                x_expression = real_number_field.gen() - k * y_expression
                return (real_number_field, x_expression, y_expression)
            else:
                k += 1
        except _IsolateFactorError:
            extra_prec += 1
            CIF = ComplexIntervalField(53 * 2 ** extra_prec)
            z_val = CIF(number_field.gen_embedding())
            x_val = z_val.real()
            y_val = z_val.imag()
    return None