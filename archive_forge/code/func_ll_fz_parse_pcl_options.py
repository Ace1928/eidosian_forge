from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
def ll_fz_parse_pcl_options(opts, args):
    """
    Low-level wrapper for `::fz_parse_pcl_options()`.
    Parse PCL options.

    Currently defined options and values are as follows:

    	preset=X	Either "generic" or one of the presets as for fz_pcl_preset.
    	spacing=0	No vertical spacing capability
    	spacing=1	PCL 3 spacing (<ESC>*p+<n>Y)
    	spacing=2	PCL 4 spacing (<ESC>*b<n>Y)
    	spacing=3	PCL 5 spacing (<ESC>*b<n>Y and clear seed row)
    	mode2		Disable/Enable mode 2 graphics compression
    	mode3		Disable/Enable mode 3 graphics compression
    	eog_reset	End of graphics (<ESC>*rB) resets all parameters
    	has_duplex	Duplex supported (<ESC>&l<duplex>S)
    	has_papersize	Papersize setting supported (<ESC>&l<sizecode>A)
    	has_copies	Number of copies supported (<ESC>&l<copies>X)
    	is_ljet4pjl	Disable/Enable HP 4PJL model-specific output
    	is_oce9050	Disable/Enable Oce 9050 model-specific output
    """
    return _mupdf.ll_fz_parse_pcl_options(opts, args)