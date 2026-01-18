import contextlib
import io
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import dns.edns
import dns.entropy
import dns.enum
import dns.exception
import dns.flags
import dns.name
import dns.opcode
import dns.rcode
import dns.rdata
import dns.rdataclass
import dns.rdatatype
import dns.rdtypes.ANY.OPT
import dns.rdtypes.ANY.TSIG
import dns.renderer
import dns.rrset
import dns.tsig
import dns.ttl
import dns.wire
def resolve_chaining(self) -> ChainingResult:
    """Follow the CNAME chain in the response to determine the answer
        RRset.

        Raises ``dns.message.NotQueryResponse`` if the message is not
        a response.

        Raises ``dns.message.ChainTooLong`` if the CNAME chain is too long.

        Raises ``dns.message.AnswerForNXDOMAIN`` if the rcode is NXDOMAIN
        but an answer was found.

        Raises ``dns.exception.FormError`` if the question count is not 1.

        Returns a ChainingResult object.
        """
    if self.flags & dns.flags.QR == 0:
        raise NotQueryResponse
    if len(self.question) != 1:
        raise dns.exception.FormError
    question = self.question[0]
    qname = question.name
    min_ttl = dns.ttl.MAX_TTL
    answer = None
    count = 0
    cnames = []
    while count < MAX_CHAIN:
        try:
            answer = self.find_rrset(self.answer, qname, question.rdclass, question.rdtype)
            min_ttl = min(min_ttl, answer.ttl)
            break
        except KeyError:
            if question.rdtype != dns.rdatatype.CNAME:
                try:
                    crrset = self.find_rrset(self.answer, qname, question.rdclass, dns.rdatatype.CNAME)
                    cnames.append(crrset)
                    min_ttl = min(min_ttl, crrset.ttl)
                    for rd in crrset:
                        qname = rd.target
                        break
                    count += 1
                    continue
                except KeyError:
                    break
            else:
                break
    if count >= MAX_CHAIN:
        raise ChainTooLong
    if self.rcode() == dns.rcode.NXDOMAIN and answer is not None:
        raise AnswerForNXDOMAIN
    if answer is None:
        auname = qname
        while True:
            try:
                srrset = self.find_rrset(self.authority, auname, question.rdclass, dns.rdatatype.SOA)
                min_ttl = min(min_ttl, srrset.ttl, srrset[0].minimum)
                break
            except KeyError:
                try:
                    auname = auname.parent()
                except dns.name.NoParent:
                    break
    return ChainingResult(qname, answer, min_ttl, cnames)