from boto.ec2.ec2object import TaggedEC2Object
@property
def status_message(self):
    return self._status.message