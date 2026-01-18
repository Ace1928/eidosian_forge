import dataclasses
import io
import math
from typing import Iterable, Optional
from ortools.math_opt.python import model
def compute_model_ranges(mdl: model.Model) -> ModelRanges:
    """Returns the ranges of the finite non-zero values in the given model.

    Args:
      mdl: The input model.

    Returns:
      The ranges of the finite non-zero values in the model.
    """
    return ModelRanges(objective_terms=absolute_finite_non_zeros_range((term.coefficient for term in mdl.objective.linear_terms())), variable_bounds=merge_optional_ranges(absolute_finite_non_zeros_range((v.lower_bound for v in mdl.variables())), absolute_finite_non_zeros_range((v.upper_bound for v in mdl.variables()))), linear_constraint_bounds=merge_optional_ranges(absolute_finite_non_zeros_range((c.lower_bound for c in mdl.linear_constraints())), absolute_finite_non_zeros_range((c.upper_bound for c in mdl.linear_constraints()))), linear_constraint_coefficients=absolute_finite_non_zeros_range((e.coefficient for e in mdl.linear_constraint_matrix_entries())))