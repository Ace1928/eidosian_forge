from zaqarclient.queues.v2 import core
def ensure_exists(self):
    """Ensures subscription exists

        This method is not race safe, the subscription could've been deleted
        right after it was called.
        """
    req, trans = self.client._request_and_transport()
    if not self.id and self.subscriber:
        subscription_data = {'subscriber': self.subscriber, 'ttl': self.ttl, 'options': self.options}
        subscription = core.subscription_create(trans, req, self.queue_name, subscription_data)
        if subscription and 'subscription_id' in subscription:
            self.id = subscription['subscription_id']
    if self.id:
        sub = core.subscription_get(trans, req, self.queue_name, self.id)
        self.subscriber = sub.get('subscriber')
        self.ttl = sub.get('ttl')
        self.options = sub.get('options')
        self.age = sub.get('age')
        self.confirmed = sub.get('confirmed')