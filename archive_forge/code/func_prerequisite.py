from typing import Any, List, Optional, Union
import dns.message
import dns.name
import dns.opcode
import dns.rdata
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.tsig
@prerequisite.setter
def prerequisite(self, v):
    self.sections[1] = v