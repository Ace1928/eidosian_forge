import re
import numpy as np
from xml.dom import minidom
def extract_pao_elements(ion, doc):
    """
    extract the different pao element of the xml file
    Input Parameters:
    -----------------
        ion (dict)
        doc (minidom.parse)
    Output Parameters:
    ------------------
        ion (dict): the following keys are added to the ion dict:
            npts
            delta
            cutoff
            data
            orbital
            projector
    """
    name_npts = doc.getElementsByTagName('npts')
    name_delta = doc.getElementsByTagName('delta')
    name_cutoff = doc.getElementsByTagName('cutoff')
    name_data = doc.getElementsByTagName('data')
    name_orbital = doc.getElementsByTagName('orbital')
    name_projector = doc.getElementsByTagName('projector')
    ion['orbital'] = []
    ion['projector'] = []
    for i in range(len(name_orbital)):
        ion['orbital'].append(extract_orbital(name_orbital[i]))
    for i in range(len(name_projector)):
        ion['projector'].append(extract_projector(name_projector[i]))
    if len(name_data) != len(name_npts):
        raise ValueError('len(name_data) != len(name_npts): {0} != {1}'.format(len(name_data), len(name_npts)))
    if len(name_data) != len(name_cutoff):
        raise ValueError('len(name_data) != len(name_cutoff): {0} != {1}'.format(len(name_data), len(name_cutoff)))
    if len(name_data) != len(name_delta):
        raise ValueError('len(name_data) != len(name_delta): {0} != {1}'.format(len(name_data), len(name_delta)))
    ion['npts'] = np.zeros(len(name_npts), dtype=int)
    ion['delta'] = np.zeros(len(name_delta), dtype=float)
    ion['cutoff'] = np.zeros(len(name_cutoff), dtype=float)
    ion['data'] = []
    for i in range(len(name_data)):
        ion['npts'][i] = get_data_elements(name_npts[i], int)
        ion['cutoff'][i] = get_data_elements(name_cutoff[i], float)
        ion['delta'][i] = get_data_elements(name_delta[i], float)
        ion['data'].append(get_data_elements(name_data[i], float).reshape(ion['npts'][i], 2))