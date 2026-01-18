from sympy.core.expr import Expr
from sympy.core.sympify import sympify
def prefix_unit(unit, prefixes):
    """
    Return a list of all units formed by unit and the given prefixes.

    You can use the predefined PREFIXES or BIN_PREFIXES, but you can also
    pass as argument a subdict of them if you do not want all prefixed units.

        >>> from sympy.physics.units.prefixes import (PREFIXES,
        ...                                                 prefix_unit)
        >>> from sympy.physics.units import m
        >>> pref = {"m": PREFIXES["m"], "c": PREFIXES["c"], "d": PREFIXES["d"]}
        >>> prefix_unit(m, pref)  # doctest: +SKIP
        [millimeter, centimeter, decimeter]
    """
    from sympy.physics.units.quantities import Quantity
    from sympy.physics.units import UnitSystem
    prefixed_units = []
    for prefix_abbr, prefix in prefixes.items():
        quantity = Quantity('%s%s' % (prefix.name, unit.name), abbrev='%s%s' % (prefix.abbrev, unit.abbrev), is_prefixed=True)
        UnitSystem._quantity_dimensional_equivalence_map_global[quantity] = unit
        UnitSystem._quantity_scale_factors_global[quantity] = (prefix.scale_factor, unit)
        prefixed_units.append(quantity)
    return prefixed_units