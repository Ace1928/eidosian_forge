import six
from pycadf import attachment
from pycadf import cadftaxonomy
from pycadf import cadftype
from pycadf import identifier
from pycadf import measurement
from pycadf import reason
from pycadf import reporterstep
from pycadf import resource
from pycadf import tag
from pycadf import timestamp
def add_reporterstep(self, step):
    """Add a Reporterstep

        :param step: Reporterstep to be added to reporterchain
        """
    if step is not None and isinstance(step, reporterstep.Reporterstep):
        if step.is_valid():
            if not hasattr(self, EVENT_KEYNAME_REPORTERCHAIN):
                setattr(self, EVENT_KEYNAME_REPORTERCHAIN, list())
            reporterchain = getattr(self, EVENT_KEYNAME_REPORTERCHAIN)
            reporterchain.append(step)
        else:
            raise ValueError('Invalid reporterstep')
    else:
        raise ValueError('Invalid reporterstep. Value must be a Reporterstep')