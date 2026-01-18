import os
import warnings
from ..base import (
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
from ...utils.filemanip import load_json, save_json, split_filename
def get_staple_args(self, ranking):
    classtype = self.inputs.classifier_type
    if classtype not in ['STAPLE', 'MV']:
        return None
    if ranking == 'ALL':
        return '-ALL'
    if not isdefined(self.inputs.template_file):
        err = "LabelFusion requires a value for input 'tramplate_file' when 'classifier_type' is set to '%s' and 'sm_ranking' is set to '%s'."
        raise NipypeInterfaceError(err % (classtype, ranking))
    if not isdefined(self.inputs.template_num):
        err = "LabelFusion requires a value for input 'template-num' when 'classifier_type' is set to '%s' and 'sm_ranking' is set to '%s'."
        raise NipypeInterfaceError(err % (classtype, ranking))
    if ranking == 'GNCC':
        if not isdefined(self.inputs.template_num):
            err = "LabelFusion requires a value for input 'template_num' when 'classifier_type' is set to '%s' and 'sm_ranking' is set to '%s'."
            raise NipypeInterfaceError(err % (classtype, ranking))
        return '-%s %d %s %s' % (ranking, self.inputs.template_num, self.inputs.file_to_seg, self.inputs.template_file)
    elif ranking == 'ROINCC':
        if not isdefined(self.inputs.dilation_roi):
            err = "LabelFusion requires a value for input 'dilation_roi' when 'classifier_type' is set to '%s' and 'sm_ranking' is set to '%s'."
            raise NipypeInterfaceError(err % (classtype, ranking))
        elif self.inputs.dilation_roi < 1:
            err = "The 'dilation_roi' trait of a LabelFusionInput instance must be an integer >= 1, but a value of '%s' was specified."
            raise NipypeInterfaceError(err % self.inputs.dilation_roi)
        return '-%s %d %d %s %s' % (ranking, self.inputs.dilation_roi, self.inputs.template_num, self.inputs.file_to_seg, self.inputs.template_file)
    elif ranking == 'LNCC':
        if not isdefined(self.inputs.kernel_size):
            err = "LabelFusion requires a value for input 'kernel_size' when 'classifier_type' is set to '%s' and 'sm_ranking' is set to '%s'."
            raise NipypeInterfaceError(err % (classtype, ranking))
        return '-%s %f %d %s %s' % (ranking, self.inputs.kernel_size, self.inputs.template_num, self.inputs.file_to_seg, self.inputs.template_file)