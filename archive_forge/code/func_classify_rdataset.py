import enum
import io
from typing import Any, Dict, Optional
import dns.immutable
import dns.name
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.renderer
import dns.rrset
@classmethod
def classify_rdataset(cls, rdataset: dns.rdataset.Rdataset) -> 'NodeKind':
    return cls.classify(rdataset.rdtype, rdataset.covers)