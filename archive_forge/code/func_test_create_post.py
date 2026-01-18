import graphene
def test_create_post():
    query_string = '\n    mutation {\n      createPost(text: "Try this out") {\n        result {\n          __typename\n        }\n      }\n    }\n    '
    schema = graphene.Schema(query=Query, mutation=Mutations)
    result = schema.execute(query_string)
    assert not result.errors
    assert result.data['createPost']['result']['__typename'] == 'Success'