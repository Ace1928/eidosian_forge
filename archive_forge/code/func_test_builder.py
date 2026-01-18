import json
from fugue import (
import cloudpickle
import fugue
from tune._utils.serialization import from_base64
from tune.constants import (
from tune.concepts.dataset import TuneDatasetBuilder, _to_trail_row, TuneDataset
from tune.concepts.space import Grid, Rand
from tune.concepts.space.spaces import Space
from tune.concepts.flow import Trial
def test_builder(tmpdir):
    space = Space(a=1, b=2, c=Grid(2, 3))
    builder = TuneDatasetBuilder(space, str(tmpdir))

    def assert_count(df: DataFrame, n: int, schema=None) -> None:
        assert len(df.as_array()) == n
        if schema is not None:
            assert df.schema == schema
    with FugueWorkflow() as dag:
        df = builder.build(dag).data
        df.show()
    dag.run()
    df1 = ArrayDataFrame([[0, 1], [1, 1], [0, 2]], 'a:int,b:int')
    with FugueWorkflow() as dag:
        builder.add_dfs(WorkflowDataFrames(x=dag.df(df1)))
        dataset = builder.build(dag)
        assert ['x'] == dataset.dfs
        assert [] == dataset.keys
        df = dataset.data
        df.show()
        df.output(assert_count, params=dict(n=2, schema=f'__tune_df__x:str,{TUNE_DATASET_TRIALS}:str'))
    dag.run()
    space = Space(b=Rand(0, 1), a=1, c=Grid(2, 3), d=Grid('a', 'b'))
    df2 = ArrayDataFrame([[0, 1], [1, 1], [3, 2]], 'a:int,bb:int')
    df3 = ArrayDataFrame([[10, 1], [11, 1], [10, 2]], 'a:int,c:int')
    builder = TuneDatasetBuilder(space)
    engine = NativeExecutionEngine(conf={TUNE_TEMP_PATH: str(tmpdir)})
    with FugueWorkflow() as dag:
        dfs = WorkflowDataFrames(a=dag.df(df1).partition_by('a'), b=dag.df(df2).partition_by('a'))
        dataset = builder.add_dfs(dfs, 'inner').add_df('c', dag.df(df3), 'cross').build(dag)
        assert ['a'] == dataset.keys
        assert ['a', 'b', 'c'] == dataset.dfs
        df = dataset.data
        df.show()
        df.output(assert_count, params=dict(n=8, schema=f'a:int,__tune_df__a:str,__tune_df__b:str,__tune_df__c:str,{TUNE_DATASET_TRIALS}:str'))
        df = builder.build(dag, batch_size=3).data
        df.show()
        df.output(assert_count, params=dict(n=4, schema=f'a:int,__tune_df__a:str,__tune_df__b:str,__tune_df__c:str,{TUNE_DATASET_TRIALS}:str'))
    dag.run(engine)