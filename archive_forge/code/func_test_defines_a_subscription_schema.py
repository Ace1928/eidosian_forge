import copy
from ..argument import Argument
from ..definitions import GrapheneGraphQLType
from ..enum import Enum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import Boolean, Int, String
from ..schema import Schema
from ..structures import List, NonNull
from ..union import Union
def test_defines_a_subscription_schema():
    blog_schema = Schema(Query, subscription=Subscription)
    assert blog_schema.subscription == Subscription
    assert blog_schema.graphql_schema.subscription_type.graphene_type == Subscription
    subscription = Subscription._meta.fields['article_subscribe']
    assert subscription.type == Article
    assert subscription.type._meta.name == 'Article'