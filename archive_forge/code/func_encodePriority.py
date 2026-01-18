import io, logging, socket, os, pickle, struct, time, re
from stat import ST_DEV, ST_INO, ST_MTIME
import queue
import threading
import copy
def encodePriority(self, facility, priority):
    """
        Encode the facility and priority. You can pass in strings or
        integers - if strings are passed, the facility_names and
        priority_names mapping dictionaries are used to convert them to
        integers.
        """
    if isinstance(facility, str):
        facility = self.facility_names[facility]
    if isinstance(priority, str):
        priority = self.priority_names[priority]
    return facility << 3 | priority