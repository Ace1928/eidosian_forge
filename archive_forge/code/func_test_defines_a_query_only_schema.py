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
def test_defines_a_query_only_schema():
    blog_schema = Schema(Query)
    assert blog_schema.query == Query
    assert blog_schema.graphql_schema.query_type.graphene_type == Query
    article_field = Query._meta.fields['article']
    assert article_field.type == Article
    assert article_field.type._meta.name == 'Article'
    article_field_type = article_field.type
    assert issubclass(article_field_type, ObjectType)
    title_field = article_field_type._meta.fields['title']
    assert title_field.type == String
    author_field = article_field_type._meta.fields['author']
    author_field_type = author_field.type
    assert issubclass(author_field_type, ObjectType)
    recent_article_field = author_field_type._meta.fields['recent_article']
    assert recent_article_field.type == Article
    feed_field = Query._meta.fields['feed']
    assert feed_field.type.of_type == Article