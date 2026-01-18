from ._cares import ffi as _ffi, lib as _lib
import _cffi_backend  # hint for bundler tools
from . import errno
from .utils import ascii_bytes, maybe_str, parse_name
from ._version import __version__
import collections.abc
import socket
import math
import functools
import sys
def parse_result(query_type, abuf, alen):
    if query_type == _lib.T_A:
        addrttls = _ffi.new('struct ares_addrttl[]', PYCARES_ADDRTTL_SIZE)
        naddrttls = _ffi.new('int*', PYCARES_ADDRTTL_SIZE)
        parse_status = _lib.ares_parse_a_reply(abuf, alen, _ffi.NULL, addrttls, naddrttls)
        if parse_status != _lib.ARES_SUCCESS:
            result = None
            status = parse_status
        else:
            result = [ares_query_a_result(addrttls[i]) for i in range(naddrttls[0])]
            status = None
    elif query_type == _lib.T_AAAA:
        addrttls = _ffi.new('struct ares_addr6ttl[]', PYCARES_ADDRTTL_SIZE)
        naddrttls = _ffi.new('int*', PYCARES_ADDRTTL_SIZE)
        parse_status = _lib.ares_parse_aaaa_reply(abuf, alen, _ffi.NULL, addrttls, naddrttls)
        if parse_status != _lib.ARES_SUCCESS:
            result = None
            status = parse_status
        else:
            result = [ares_query_aaaa_result(addrttls[i]) for i in range(naddrttls[0])]
            status = None
    elif query_type == _lib.T_CAA:
        caa_reply = _ffi.new('struct ares_caa_reply **')
        parse_status = _lib.ares_parse_caa_reply(abuf, alen, caa_reply)
        if parse_status != _lib.ARES_SUCCESS:
            result = None
            status = parse_status
        else:
            result = []
            caa_reply_ptr = caa_reply[0]
            while caa_reply_ptr != _ffi.NULL:
                result.append(ares_query_caa_result(caa_reply_ptr))
                caa_reply_ptr = caa_reply_ptr.next
            _lib.ares_free_data(caa_reply[0])
            status = None
    elif query_type == _lib.T_CNAME:
        host = _ffi.new('struct hostent **')
        parse_status = _lib.ares_parse_a_reply(abuf, alen, host, _ffi.NULL, _ffi.NULL)
        if parse_status != _lib.ARES_SUCCESS:
            result = None
            status = parse_status
        else:
            result = ares_query_cname_result(host[0])
            _lib.ares_free_hostent(host[0])
            status = None
    elif query_type == _lib.T_MX:
        mx_reply = _ffi.new('struct ares_mx_reply **')
        parse_status = _lib.ares_parse_mx_reply(abuf, alen, mx_reply)
        if parse_status != _lib.ARES_SUCCESS:
            result = None
            status = parse_status
        else:
            result = []
            mx_reply_ptr = mx_reply[0]
            while mx_reply_ptr != _ffi.NULL:
                result.append(ares_query_mx_result(mx_reply_ptr))
                mx_reply_ptr = mx_reply_ptr.next
            _lib.ares_free_data(mx_reply[0])
            status = None
    elif query_type == _lib.T_NAPTR:
        naptr_reply = _ffi.new('struct ares_naptr_reply **')
        parse_status = _lib.ares_parse_naptr_reply(abuf, alen, naptr_reply)
        if parse_status != _lib.ARES_SUCCESS:
            result = None
            status = parse_status
        else:
            result = []
            naptr_reply_ptr = naptr_reply[0]
            while naptr_reply_ptr != _ffi.NULL:
                result.append(ares_query_naptr_result(naptr_reply_ptr))
                naptr_reply_ptr = naptr_reply_ptr.next
            _lib.ares_free_data(naptr_reply[0])
            status = None
    elif query_type == _lib.T_NS:
        hostent = _ffi.new('struct hostent **')
        parse_status = _lib.ares_parse_ns_reply(abuf, alen, hostent)
        if parse_status != _lib.ARES_SUCCESS:
            result = None
            status = parse_status
        else:
            result = []
            host = hostent[0]
            i = 0
            while host.h_aliases[i] != _ffi.NULL:
                result.append(ares_query_ns_result(host.h_aliases[i]))
                i += 1
            _lib.ares_free_hostent(host)
            status = None
    elif query_type == _lib.T_PTR:
        hostent = _ffi.new('struct hostent **')
        parse_status = _lib.ares_parse_ptr_reply(abuf, alen, _ffi.NULL, 0, socket.AF_UNSPEC, hostent)
        if parse_status != _lib.ARES_SUCCESS:
            result = None
            status = parse_status
        else:
            aliases = []
            host = hostent[0]
            i = 0
            while host.h_aliases[i] != _ffi.NULL:
                aliases.append(maybe_str(_ffi.string(host.h_aliases[i])))
                i += 1
            result = ares_query_ptr_result(host, aliases)
            _lib.ares_free_hostent(host)
            status = None
    elif query_type == _lib.T_SOA:
        soa_reply = _ffi.new('struct ares_soa_reply **')
        parse_status = _lib.ares_parse_soa_reply(abuf, alen, soa_reply)
        if parse_status != _lib.ARES_SUCCESS:
            result = None
            status = parse_status
        else:
            result = ares_query_soa_result(soa_reply[0])
            _lib.ares_free_data(soa_reply[0])
            status = None
    elif query_type == _lib.T_SRV:
        srv_reply = _ffi.new('struct ares_srv_reply **')
        parse_status = _lib.ares_parse_srv_reply(abuf, alen, srv_reply)
        if parse_status != _lib.ARES_SUCCESS:
            result = None
            status = parse_status
        else:
            result = []
            srv_reply_ptr = srv_reply[0]
            while srv_reply_ptr != _ffi.NULL:
                result.append(ares_query_srv_result(srv_reply_ptr))
                srv_reply_ptr = srv_reply_ptr.next
            _lib.ares_free_data(srv_reply[0])
            status = None
    elif query_type == _lib.T_TXT:
        txt_reply = _ffi.new('struct ares_txt_ext **')
        parse_status = _lib.ares_parse_txt_reply_ext(abuf, alen, txt_reply)
        if parse_status != _lib.ARES_SUCCESS:
            result = None
            status = parse_status
        else:
            result = []
            txt_reply_ptr = txt_reply[0]
            tmp_obj = None
            while True:
                if txt_reply_ptr == _ffi.NULL:
                    if tmp_obj is not None:
                        result.append(ares_query_txt_result(tmp_obj))
                    break
                if txt_reply_ptr.record_start == 1:
                    if tmp_obj is not None:
                        result.append(ares_query_txt_result(tmp_obj))
                    tmp_obj = ares_query_txt_result_chunk(txt_reply_ptr)
                else:
                    new_chunk = ares_query_txt_result_chunk(txt_reply_ptr)
                    tmp_obj.text += new_chunk.text
                txt_reply_ptr = txt_reply_ptr.next
            _lib.ares_free_data(txt_reply[0])
            status = None
    else:
        raise ValueError('invalid query type specified')
    return (result, status)