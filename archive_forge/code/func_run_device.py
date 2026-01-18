from zmq import PUB
from zmq.devices.monitoredqueue import monitored_queue
from zmq.devices.proxydevice import ProcessProxy, Proxy, ProxyBase, ThreadProxy
def run_device(self):
    ins, outs, mons = self._setup_sockets()
    monitored_queue(ins, outs, mons, self._in_prefix, self._out_prefix)