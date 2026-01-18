import os.path as op
import inspect
import numpy as np
from ... import logging
from ..base import (
def dipy_to_nipype_interface(cls_name, dipy_flow, BaseClass=DipyBaseInterface):
    """Construct a class in order to respect nipype interface specifications.

    This convenient class factory convert a DIPY Workflow to a nipype
    interface.

    Parameters
    ----------
    cls_name: string
        new class name
    dipy_flow: Workflow class type.
        It should be any children class of `dipy.workflows.workflow.Worflow`
    BaseClass: object
        nipype instance object

    Returns
    -------
    newclass: object
        new nipype interface specification class

    """
    parser = IntrospectiveArgumentParser()
    flow = dipy_flow()
    parser.add_workflow(flow)
    default_values = list(get_default_args(flow.run).values())
    optional_params = [args + (val,) for args, val in zip(parser.optional_parameters, default_values)]
    start = len(parser.optional_parameters) - len(parser.output_parameters)
    output_parameters = [args + (val,) for args, val in zip(parser.output_parameters, default_values[start:])]
    input_parameters = parser.positional_parameters + optional_params
    input_spec = create_interface_specs('{}InputSpec'.format(cls_name), input_parameters, BaseClass=BaseInterfaceInputSpec)
    output_spec = create_interface_specs('{}OutputSpec'.format(cls_name), output_parameters, BaseClass=TraitedSpec)

    def _run_interface(self, runtime):
        flow = dipy_flow()
        args = self.inputs.get()
        flow.run(**args)

    def _list_outputs(self):
        outputs = self._outputs().get()
        out_dir = outputs.get('out_dir', '.')
        for key, values in outputs.items():
            outputs[key] = op.join(out_dir, values)
        return outputs
    newclass = type(str(cls_name), (BaseClass,), {'input_spec': input_spec, 'output_spec': output_spec, '_run_interface': _run_interface, '_list_outputs:': _list_outputs})
    return newclass