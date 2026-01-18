from pyomo.common.collections import ComponentMap
from pyomo.core.base import (
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.core.base import TransformationFactory
from pyomo.core.base.suffix import SuffixFinder
from pyomo.core.expr import replace_expressions
from pyomo.util.components import rename_components
def propagate_solution(self, scaled_model, original_model):
    """This method takes the solution in scaled_model and maps it back to
        the original model.

        It will also transform duals and reduced costs if the suffixes
        'dual' and/or 'rc' are present.  The :code:`scaled_model`
        argument must be a model that was already scaled using this
        transformation as it expects data from the transformation to
        perform the back mapping.

        Parameters
        ----------
        scaled_model : Pyomo Model
           The model that was previously scaled with this transformation
        original_model : Pyomo Model
           The original unscaled source model

        """
    if not hasattr(scaled_model, 'component_scaling_factor_map'):
        raise AttributeError('ScaleModel:propagate_solution called with scaled_model that does not have a component_scaling_factor_map. It is possible this method was called using a model that was not scaled with the ScaleModel transformation')
    if not hasattr(scaled_model, 'scaled_component_to_original_name_map'):
        raise AttributeError('ScaleModel:propagate_solution called with scaled_model that does not have a scaled_component_to_original_name_map. It is possible this method was called using a model that was not scaled with the ScaleModel transformation')
    component_scaling_factor_map = scaled_model.component_scaling_factor_map
    scaled_component_to_original_name_map = scaled_model.scaled_component_to_original_name_map
    check_reduced_costs = type(scaled_model.component('rc')) is Suffix
    check_dual = type(scaled_model.component('dual')) is Suffix and type(original_model.component('dual')) is Suffix
    if check_reduced_costs or check_dual:
        scaled_objectives = list(scaled_model.component_data_objects(ctype=Objective, active=True, descend_into=True))
        if len(scaled_objectives) != 1:
            raise NotImplementedError('ScaleModel.propagate_solution requires a single active objective function, but %d objectives found.' % len(scaled_objectives))
        else:
            objective_scaling_factor = component_scaling_factor_map[scaled_objectives[0]]
    for scaled_v in scaled_model.component_objects(ctype=Var, descend_into=True):
        original_v_path = scaled_component_to_original_name_map[scaled_v]
        original_v = original_model.find_component(original_v_path)
        for k in scaled_v:
            original_v[k].set_value(value(scaled_v[k]) / component_scaling_factor_map[scaled_v[k]], skip_validation=True)
            if check_reduced_costs and scaled_v[k] in scaled_model.rc:
                original_model.rc[original_v[k]] = scaled_model.rc[scaled_v[k]] * component_scaling_factor_map[scaled_v[k]] / objective_scaling_factor
    if check_dual:
        for scaled_c in scaled_model.component_objects(ctype=Constraint, descend_into=True):
            original_c = original_model.find_component(scaled_component_to_original_name_map[scaled_c])
            for k in scaled_c:
                original_model.dual[original_c[k]] = scaled_model.dual[scaled_c[k]] * component_scaling_factor_map[scaled_c[k]] / objective_scaling_factor