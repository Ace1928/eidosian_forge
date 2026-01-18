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
def fz_pcl_preset(self, preset):
    """
        Class-aware wrapper for `::fz_pcl_preset()`.
        	Initialize PCL option struct for a given preset.

        	Currently defined presets include:

        		generic	Generic PCL printer
        		ljet4	HP DeskJet
        		dj500	HP DeskJet 500
        		fs600	Kyocera FS-600
        		lj	HP LaserJet, HP LaserJet Plus
        		lj2	HP LaserJet IIp, HP LaserJet IId
        		lj3	HP LaserJet III
        		lj3d	HP LaserJet IIId
        		lj4	HP LaserJet 4
        		lj4pl	HP LaserJet 4 PL
        		lj4d	HP LaserJet 4d
        		lp2563b	HP 2563B line printer
        		oce9050	Oce 9050 Line printer
        """
    return _mupdf.FzPclOptions_fz_pcl_preset(self, preset)